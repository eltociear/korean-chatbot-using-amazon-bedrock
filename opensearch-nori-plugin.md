# RAG를 이용하여 Nori 언어 분석기를 이용한 OpenSearch 성능향상

[2023년 10월에 한국어, 일본어, 중국어에 대한 새로운 언어 분석기 플러그인이 OpenSearch에 추가](https://aws.amazon.com/ko/about-aws/whats-new/2023/10/amazon-opensearch-four-language-analyzers/) 되었습니다. 이제 한국어는 Nori 분석기를 이용하여 한국에 맞는 RAG 검색을 할 수 있습니다. 여기서는 OpenSearch에서 [Nori 플러그인을 이용한 한국어 분석](https://aws.amazon.com/ko/blogs/tech/amazon-opensearch-service-korean-nori-plugin-for-analysis/) 블로그를 참조하여 OpenSearch의 한국어 분석기능을 향상시키고자 합니다. 

### 설치

[Nori 설치](https://esbook.kimjmin.net/06-text-analysis/6.7-stemming/6.7.2-nori#nori-1)와 같이 설치할 수 있습니다.

```text
$ bin/elasticsearch-plugin install analysis-nori
```

[dockerfile에 반영](https://awstip.com/aws-cdk-logstash-setup-with-connection-to-opensearch-2f99ba6c3053)을 참조하여 설정합니다.

```text
RUN logstash-plugin install logstash-output-opensearch
```

### 실행결과

검색 결과는 포맷은 아래와 같습니다. 
```java
{
"hits":{
   "total":{
      "value":143,
      "relation":"eq"
   },
   "max_score":3.1938949,
   "hits":[
      {
         "_index":"rag-index-upload-docs_2023-hi-tech_sds_통합보안센터_siem_on_amazon_opensearch_service_sds_(한글)_2.pdf",
         "_id":"a4244b50-742d-45f8-acc5-12883df2dc58",
         "_score":3.1938949,
         "_source":{
            "vector_field":[0.48046875,-0.19335938, ...... , -0.37304688],
            "text":". 보안 위협 모델을 사용하여 위험 파악 및 우선순위 지정보안 전문가들은 어떻게 그들의 시스템을 안전하게 보호하나?위협 모델링 구축 질문: 공격자는 누구인가? 그들이 가지고 있는 도구나 능력은 무엇인가? 그들이 우리를 대상으로 하고자 하는 것은 무엇인가?  답변 예제: 계정 탈취 암호 추측 및 나열 계정 탈취를 목적으로 스팸 발송 새로운 보안 서비스 및 기능을 정기적으로 평가 및 구현정기 검토 계획: 규정 준수 요구 사항, 새 AWS 보안 기능/서비스 평가, 업계 최신 소식 확인 등의 검토 활동 일정을 생성합니다. AWS 서비스 및 기능 검색: 사용 중인 서비스에 적용 가능한 보안 기능을 검색하고 새로 릴리스되는 기능을 검토합니다. AWS 서비스 온보딩 프로세스 정의: 새 AWS 서비스 온보딩을 위한 프로세스를 정의합니다. 이 과정에서 새 AWS 서비스의 기능과 워크로드의 규정 준수 요구 사항을 평가할 방법도 정의합니다. 새로운 서비스 및 기능 테스트: 프로덕션 환경을 거의 동일하게 복제한 비프로덕션 환경에서, 새로 릴리스하는 서비스 및 기능을 테스트합니다. 기타 방어 메커니즘 구현: 워크로드를 방어하기 위한 자동화된 메커니즘을 구현하고 사용 가능한 옵션을 살펴봅니다. SEC2. 사람과 시스템에 대한 자격 증명은 어떻게 \t    관리하십니까?? 강력한 로그인 메커니즘 사용 임시 자격 증명 사용 안전하게 보안 암호 저장 및 사용 중앙 집중식 자격 증명 공급자 사용 정기적으로 자격 증명 감사 및 교체 사용자 그룹 및 속성 활용SEC2. 모범 사례 강력한 로그인 메커니즘 사용MFA 로그인을 적용하는 IAM 정책 생성: 사용자가 내 보안 자격 증명 페이지에서 역할을 수임하고 자신의 자격 증명을 변경하고 MFA 디바이스를 관리할 수 있도록 하는 몇 가지 작업을 제외한 모든 IAM 작업을 금지하는 고객 관리형 IAM 정책을 생성합니다",
            "metadata":{
               "name":"docs/2023-Hi-Tech/SDS/통합보안센터/SIEM on Amazon Opensearch Service_SDS (한글)_2.pdf",
               "uri":"https://d1gbc5k2u14y8r.cloudfront.net/docs/2023-Hi-Tech/SDS/%E1%84%90%E1%85%A9%E1%86%BC%E1%84%92%E1%85%A1%E1%86%B8%E1%84%87%E1%85%A9%E1%84%8B%E1%85%A1%E1%86%AB%E1%84%89%E1%85%A6%E1%86%AB%E1%84%90%E1%85%A5/SIEM%20on%20Amazon%20Opensearch%20Service_SDS%20%28%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%B3%E1%86%AF%29_2.pdf"
            }
         }
      },
}
```

## Reference

[Korean (nori) Analysis Plugin](https://www.elastic.co/guide/en/elasticsearch/plugins/7.10/analysis-nori.html)

[Plugins by engine version in Amazon OpenSearch Service](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/supported-plugins.html)

[노리 (nori) 한글 형태소 분석기](https://esbook.kimjmin.net/06-text-analysis/6.7-stemming/6.7.2-nori)

[Elasticsearch를 검색 엔진으로 사용하기(1): Nori 한글 형태소 분석기로 검색 고도화 하기](https://hanamon.kr/elasticsearch-%EA%B2%80%EC%83%89%EC%97%94%EC%A7%84-nori-%ED%98%95%ED%83%9C%EC%86%8C-%EB%B6%84%EC%84%9D%EA%B8%B0-%EA%B2%80%EC%83%89-%EA%B3%A0%EB%8F%84%ED%99%94-%EB%B0%A9%EB%B2%95/)
